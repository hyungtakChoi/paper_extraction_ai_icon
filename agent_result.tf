
# ============================================================
# HIPAA-Compliant AWS Infrastructure
# Service  : 논문 분석 핵심 추출 AI (BERT + Transformer)
# Framework: TensorFlow | Region: us-east-1
# Tags     : project=ai-infra, environment=production
# ============================================================

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = "us-east-1"
}

locals {
  tags = {
    project     = "ai-infra"
    environment = "production"
    compliance  = "HIPAA"
  }
}

# ──────────────────────────────────────────
# 1. VPC & Networking
# ──────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = merge(local.tags, { Name = "hipaa-vpc" })
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = false
  tags                    = merge(local.tags, { Name = "public-subnet" })
}

resource "aws_subnet" "private_a" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.10.0/24"
  availability_zone = "us-east-1a"
  tags              = merge(local.tags, { Name = "private-subnet-a" })
}

resource "aws_subnet" "private_b" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.11.0/24"
  availability_zone = "us-east-1b"
  tags              = merge(local.tags, { Name = "private-subnet-b" })
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.tags, { Name = "hipaa-igw" })
}

resource "aws_eip" "nat" { domain = "vpc" }

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public.id
  tags          = merge(local.tags, { Name = "hipaa-nat" })
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }
  tags = merge(local.tags, { Name = "private-rt" })
}

resource "aws_route_table_association" "private_a" {
  subnet_id      = aws_subnet.private_a.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "private_b" {
  subnet_id      = aws_subnet.private_b.id
  route_table_id = aws_route_table.private.id
}

# ──────────────────────────────────────────
# 2. KMS (암호화 키 — HIPAA §164.312(a)(2)(iv))
# ──────────────────────────────────────────
resource "aws_kms_key" "hipaa" {
  description             = "HIPAA KMS key for paper-ai"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  tags                    = local.tags
}

resource "aws_kms_alias" "hipaa" {
  name          = "alias/hipaa-paper-ai"
  target_key_id = aws_kms_key.hipaa.key_id
}

# ──────────────────────────────────────────
# 3. S3 (논문 데이터 저장 — KMS 암호화)
# ──────────────────────────────────────────
resource "aws_s3_bucket" "papers" {
  bucket        = "hipaa-paper-ai-data-${random_id.suffix.hex}"
  force_destroy = false
  tags          = local.tags
}

resource "random_id" "suffix" { byte_length = 4 }

resource "aws_s3_bucket_versioning" "papers" {
  bucket = aws_s3_bucket.papers.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "papers" {
  bucket = aws_s3_bucket.papers.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.hipaa.arn
    }
  }
}

resource "aws_s3_bucket_public_access_block" "papers" {
  bucket                  = aws_s3_bucket.papers.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ──────────────────────────────────────────
# 4. RDS Aurora (결과 저장 — 암호화)
# ──────────────────────────────────────────
resource "aws_db_subnet_group" "aurora" {
  name       = "hipaa-aurora-subnet"
  subnet_ids = [aws_subnet.private_a.id, aws_subnet.private_b.id]
  tags       = local.tags
}

resource "aws_security_group" "aurora" {
  name   = "hipaa-aurora-sg"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }
  egress { from_port = 0; to_port = 0; protocol = "-1"; cidr_blocks = ["0.0.0.0/0"] }
  tags = local.tags
}

resource "aws_rds_cluster" "aurora" {
  cluster_identifier      = "hipaa-paper-ai-db"
  engine                  = "aurora-postgresql"
  engine_version          = "15.4"
  database_name           = "paperai"
  master_username         = "admin"
  master_password         = aws_secretsmanager_secret_version.db_pass.secret_string
  db_subnet_group_name    = aws_db_subnet_group.aurora.name
  vpc_security_group_ids  = [aws_security_group.aurora.id]
  storage_encrypted       = true
  kms_key_id              = aws_kms_key.hipaa.arn
  deletion_protection     = true
  backup_retention_period = 35
  skip_final_snapshot     = false
  final_snapshot_identifier = "hipaa-paper-ai-final"
  tags                    = local.tags
}

resource "aws_rds_cluster_instance" "aurora" {
  count              = 2
  identifier         = "hipaa-paper-ai-db-${count.index}"
  cluster_identifier = aws_rds_cluster.aurora.id
  instance_class     = "db.r6g.large"
  engine             = aws_rds_cluster.aurora.engine
  tags               = local.tags
}

# ──────────────────────────────────────────
# 5. Secrets Manager (자격증명 관리)
# ──────────────────────────────────────────
resource "aws_secretsmanager_secret" "db_password" {
  name       = "hipaa/paper-ai/db-password"
  kms_key_id = aws_kms_key.hipaa.arn
  tags       = local.tags
}

resource "aws_secretsmanager_secret_version" "db_pass" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = "REPLACE_WITH_SECURE_PASSWORD"
}

# ──────────────────────────────────────────
# 6. ECR (BERT/Transformer 컨테이너 이미지)
# ──────────────────────────────────────────
resource "aws_ecr_repository" "paper_ai" {
  name                 = "hipaa-paper-ai"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.hipaa.arn
  }
  tags = local.tags
}

# ──────────────────────────────────────────
# 7. ECS (GPU 추론 서버 — g5.xlarge)
# ──────────────────────────────────────────
resource "aws_security_group" "ecs" {
  name   = "hipaa-ecs-sg"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
  egress { from_port = 0; to_port = 0; protocol = "-1"; cidr_blocks = ["0.0.0.0/0"] }
  tags = local.tags
}

resource "aws_ecs_cluster" "paper_ai" {
  name = "hipaa-paper-ai-cluster"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = local.tags
}

resource "aws_iam_role" "ecs_task" {
  name = "hipaa-ecs-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_exec" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_ecs_task_definition" "bert_inference" {
  family                   = "hipaa-bert-inference"
  requires_compatibilities = ["EC2"]
  network_mode             = "awsvpc"
  cpu                      = "4096"
  memory                   = "16384"
  task_role_arn            = aws_iam_role.ecs_task.arn
  execution_role_arn       = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = "bert-inference"
    image     = "${aws_ecr_repository.paper_ai.repository_url}:latest"
    essential = true
    portMappings = [{ containerPort = 8080, protocol = "tcp" }]
    environment = [
      { name = "MODEL_TYPE", value = "BERT" },
      { name = "FRAMEWORK",  value = "TensorFlow" }
    ]
    secrets = [{
      name      = "DB_PASSWORD"
      valueFrom = aws_secretsmanager_secret.db_password.arn
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/hipaa-bert-inference"
        "awslogs-region"        = "us-east-1"
        "awslogs-stream-prefix" = "ecs"
      }
    }
    resourceRequirements = [{
      type  = "GPU"
      value = "1"
    }]
  }])
  tags = local.tags
}

# ──────────────────────────────────────────
# 8. WAF v2 (HIPAA §164.312(e)(1))
# ──────────────────────────────────────────
resource "aws_wafv2_web_acl" "hipaa" {
  name  = "hipaa-paper-ai-waf"
  scope = "REGIONAL"

  default_action { allow {} }

  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1
    override_action { none {} }
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSet"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 2
    override_action { none {} }
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "KnownBadInputs"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "hipaa-waf"
    sampled_requests_enabled   = true
  }
  tags = local.tags
}

# ──────────────────────────────────────────
# 9. Cognito (MFA 필수 — HIPAA §164.312(d))
# ──────────────────────────────────────────
resource "aws_cognito_user_pool" "paper_ai" {
  name = "hipaa-paper-ai-pool"

  mfa_configuration = "ON"
  software_token_mfa_configuration { enabled = true }

  password_policy {
    minimum_length                   = 14
    require_uppercase                = true
    require_lowercase                = true
    require_numbers                  = true
    require_symbols                  = true
    temporary_password_validity_days = 1
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }
  tags = local.tags
}

resource "aws_cognito_user_pool_client" "paper_ai" {
  name         = "hipaa-paper-ai-client"
  user_pool_id = aws_cognito_user_pool.paper_ai.id
  explicit_auth_flows = [
    "ALLOW_USER_SRP_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH"
  ]
}

# ──────────────────────────────────────────
# 10. CloudTrail (감사 로그 — HIPAA §164.312(b))
# ──────────────────────────────────────────
resource "aws_s3_bucket" "cloudtrail" {
  bucket        = "hipaa-paper-ai-cloudtrail-${random_id.suffix.hex}"
  force_destroy = false
  tags          = local.tags
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.hipaa.arn
    }
  }
}

resource "aws_s3_bucket_public_access_block" "cloudtrail" {
  bucket                  = aws_s3_bucket.cloudtrail.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_cloudtrail" "hipaa" {
  name                          = "hipaa-paper-ai-trail"
  s3_bucket_name                = aws_s3_bucket.cloudtrail.id
  include_global_service_events = true
  is_multi_region_trail         = true
  enable_log_file_validation    = true
  kms_key_id                    = aws_kms_key.hipaa.arn
  tags                          = local.tags
}

# ──────────────────────────────────────────
# 11. CloudWatch (모니터링 & 알람)
# ──────────────────────────────────────────
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/hipaa-bert-inference"
  retention_in_days = 365
  kms_key_id        = aws_kms_key.hipaa.arn
  tags              = local.tags
}

resource "aws_cloudwatch_metric_alarm" "gpu_utilization" {
  alarm_name          = "hipaa-gpu-high-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "GPUUtilization"
  namespace           = "AWS/ECS"
  period              = 300
  statistic           = "Average"
  threshold           = 85
  alarm_description   = "GPU utilization > 85%"
  tags                = local.tags
}

# ──────────────────────────────────────────
# Outputs
# ──────────────────────────────────────────
output "vpc_id"            { value = aws_vpc.main.id }
output "ecr_repo_url"      { value = aws_ecr_repository.paper_ai.repository_url }
output "aurora_endpoint"   { value = aws_rds_cluster.aurora.endpoint }
output "s3_bucket"         { value = aws_s3_bucket.papers.bucket }
output "cognito_pool_id"   { value = aws_cognito_user_pool.paper_ai.id }
output "kms_key_arn"       { value = aws_kms_key.hipaa.arn }
